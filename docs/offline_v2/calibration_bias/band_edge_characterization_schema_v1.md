# Band-edge instrumentation schema v1 (FROZEN)

Claude B, branch `experiment/offline-calibration-bias-v4`. Exact per-episode fields A must record for the
band-edge experiment so each competing failure mechanism can be excluded. Machine form:
`offline_v2/calibration_bias/band_edge_instrumentation.py` (+ `..._config_v1.json`). Validator:
`band_edge_instrumentation.validate_record`. **No field is model-legal**; residual/actual/nominal bias are
secret/audit-only. Mirrors Claude C's field-by-field feasibility audit (all six implementable from signals
that already exist in `ik_joint_adapter.solve()`, per-step telemetry, and skill history; collision needs a
sensor enabled).

## 5.1 Joint-limit margin
| field | dtype | unit | range | definition |
|---|---|---|---|---|
| `minimum_joint_limit_margin_rad` | float | rad | [0, joint_range] | per-episode min over ALL IK steps and 7 arm joints of `min(q−q_lower, q_upper−q)` vs `soft_joint_pos_limits` |
| `minimum_joint_limit_margin_normalized` | float | unitless | [0, 1] | same margin ÷ (q_upper − q_lower) of that joint |
| `minimum_joint_limit_joint_index` | int | index | [0, 6] | arm joint achieving the episode-min |
| `minimum_joint_limit_phase` | str | phase | PHASES | skill phase of the episode-min |

## 5.2 IK failure  (`solve().success == false` counts as one failure)
`ik_solve_count` (int), `ik_failure_count` (int), `max_consecutive_ik_failures` (int),
`ik_failure_phase_counts` (dict phase→int).

## 5.3 Two DISTINCT clamp counters (never merged)
- `joint_step_clamp_count` — raw IK step exceeded `max_joint_step` (step clamp).
- `joint_limit_clamp_count` — after the step clamp, `q_des` still exceeded the soft joint limits (limit clamp).

## 5.4 Failure phase
- `failure_phase` — last active skill state before `FAILED` (ARC_TO_FACE / MOVE_TO_PRE_GRASP / APPROACH /
  CLOSE_GRIPPER / PULL / SETTLE / RELEASE; `NONE` on success).
- `failure_transition` — the terminal transition (e.g. `APPROACH->FAILED`).

## 5.5 True handle error at CLOSE_GRIPPER → PULL (same control step)
`true_handle_error_at_close_3d` (float m), `true_handle_error_at_close_local_y` (float m, signed),
`tcp_pose_at_close` (len-7 pos+quat), `true_handle_pose_at_close` (len-7). Uses the **true** sim handle
pose, **not** the bias-perturbed perceived pose.

## 5.6 Gripper at close (same instant)
`gripper_width_at_close` (float m), `gripper_command_at_close` (float m).

## 6. Collision / contact (concrete, enabled)
`max_unintended_contact_force_N` (float N), `unintended_contact_frame_count` (int),
`first_unintended_contact_phase` (str), `contact_force_by_phase` (dict phase→N),
`contact_sensor_available` (bool). Sensor = Isaac Lab / PhysX ContactSensor on arm/hand vs cabinet,
excluding finger↔handle. Sensor-validation smoke (clean baseline / intentional-contact positive control /
discrimination) must pass before the run. Numeric threshold chosen from band data (post-run), never from
confirmatory data.

## Secret / audit-only bias fields (a model reads NONE)
`nominal_bias_y`, `residual_bias_y`, `actual_bias_y` — recorded in the audit/secret block; the validator
**rejects** any of `bias`/`residual`/`actual_bias`/`nominal_bias`/`secret` appearing inside `x`.

## Mechanism-exclusion coverage
| mechanism | excluded by | status |
|---|---|---|
| joint-limit failure | margin→0 + `joint_limit_clamp_count`>0 | covered by 5.1/5.3 |
| IK unreachability | `ik_failure_count`>0 / step-clamp saturating | covered by 5.2/5.3 |
| cabinet collision | `max_unintended_contact_force_N` / signed distance | covered by §6 (sensor enabled) |
| empty grasp (grasped air) | `gripper_width_at_close`≈open + `true_handle_error_at_close` large | covered by 5.5/5.4 |
| off-centre grasp | `true_handle_error_at_close` large but bounded | covered by 5.5 |
| PULL slip | `handle_detached` + `handle_relative_error` + `failure_phase==PULL` | already in `y` |

## Validation
`validate_record(record)` checks: all six groups present with correct dtype; non-negativity on
count/force/margin fields; normalized margin ∈ [0,1]; secret bias fields present in the audit block; no
privileged key inside `x`. Covered by `tests/offline_v2/calibration_bias/test_band_edge_instrumentation.py`.
