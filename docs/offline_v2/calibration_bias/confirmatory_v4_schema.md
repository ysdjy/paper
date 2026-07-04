# Confirmatory v4 — outcome / history / feature schema (frozen)

Claude B. Freezes and references the exact runtime schema, the model-legal history allowlist, and the
candidate feature block. Source of truth: `deployment_calibration/contracts/episode_schema_v2.py` and
`deployment_calibration/models_v2/features.py` (both power-verified). Machine form: `confirmatory_v4_config.json`.

## 1. Primary success (frozen)
- `y.success` from the frozen runtime `open_drawer` success condition (`episode_schema_v2.py`).
- `TARGET_TOLERANCE = 0.02`; `reach_timeout = pull_timeout = 16.0 s`; `close_duration 1.0 s`,
  `settle_duration 0.5 s` (`THETA_FIXED`).
- `TARGET_OPEN_LEVELS = (0.12, 0.20, 0.28)` — the task target `g.target_open_position` per block.

## 2. Failure reasons (frozen list — none deleted)
```
FAILURE_REASONS = [NONE, REACH_TIMEOUT, PULL_TIMEOUT, POSITION_TIMEOUT,
                   HANDLE_DETACHED, INVALID_PARAM, EPISODE_EXCEPTION, OTHER]
```
- `REACH_TIMEOUT` = APPROACH failure; `PULL_TIMEOUT` = PULL failure; `POSITION_TIMEOUT`, `HANDLE_DETACHED`
  retained. `DRAWER_OPEN_TIMEOUT`/`DRAWER_CLOSE_TIMEOUT` normalize to `PULL_TIMEOUT` (`normalize_failure`).
- Observed in the 306 exploration run: `NONE` 211, `POSITION_TIMEOUT` 62, `HANDLE_DETACHED` 33 (REACH/PULL
  defined but not triggered there — still retained for confirmatory).
- **Legitimate task failures (timeouts / detached / approach / pull / offset-induced) are kept as failures**,
  NOT invalidated. `INVALID_PARAM`, `EPISODE_EXCEPTION` route to technical invalidation (see analysis plan §17).

## 3. Continuous secondary outcomes (frozen)
`task_outcome_error`, `final_joint_position`, `skill_elapsed_time`, `true_handle_error_at_close_3d`,
`true_handle_error_at_close_local_y`, `gripper_width_at_close`, `phase_durations`. Regression heads are
clipped: `error_clip=[0.0,0.30]`, `time_clip=[0.0,60.0]` (`offline_v2/utility.py`).

## 4. Episode roles (probe vs candidate are DISTINCT trials)
`episode_schema_v2.py`: `ROLE_PROBE="probe"`, `ROLE_CANDIDATE="candidate"`; `history_cutoff` = number of probe
results legally visible before a decision. A session = 1 probe trial (role=probe, offset −0.04) + 3 candidate
trials (role=candidate, offsets −0.04/0/+0.04). The probe (role=probe) and the candidate at −0.04
(role=candidate) are **separate full-task executions**; the probe outcome is recorded in `H` but is **never**
used as the candidate label.

## 5. History allowlist — EXACTLY `features.probe_vector` (8 fields, in order)
```
1 theta.grasp_offset_local_y
2 theta.max_pos_step
3 theta.pull_lead
4 success
5 task_outcome_error
6 skill_elapsed_time
7 pull_phase_duration
8 final_joint_position
```
- `features.PROBE_KEYS` / `PROBE_DIM == 8`. **FIX1 (minor 7.2): all 8 fields are REQUIRED** on every probe
  HistoryEntry. If any is missing at feature-extraction time the trial is **`TECHNICAL_INVALID_SCHEMA`** — it
  must **NOT** be silently zero-filled. (`features.probe_vector` uses `h.get(field, 0.0)` internally, so the
  generator/validator must assert presence of all 8 fields BEFORE feature extraction; a runtime schema guard
  is required.) `HistoryEntry` also carries `g`, `failure_reason`, `handle_relative_error`, `mechanism_id`,
  `probe_index`, which `probe_vector` does **not** read, so the model ignores them.
- A test asserts `HISTORY_ALLOWLIST == features.PROBE_KEYS` and that a missing field →
  `TECHNICAL_INVALID_SCHEMA` (not zero-fill).

## 6. Candidate static features — `features.static_features` (8 dims)
```
theta.grasp_offset_local_y, theta.max_pos_step, theta.pull_lead,
g.target_open_position, g.target_tolerance,
x.initial_mechanism_joint_pos, x.gripper_width, (x.member == "sektion_cabinet") as {0,1}
```
`STATIC_DIM == 8`. These are the only candidate-decision inputs a deployable model may read.

## 7. Secret denylist (never a model input; audit/oracle only)
`nominal_bias, residual_bias, actual_bias, eff_signed, abs_eff, nuisance_block_id, block_seed, residual_seed,
split_identity, future_candidate_outcomes, oracle_action, hidden_state_class_label, secret_deployment_state,
hidden_state_id, damping_eff, damping_post`. `episode_schema_v2.py` isolates `secret_deployment_state` /
`hidden_state_id` as audit-only fields; the feature cache must exclude them.

## 8. Observable x (frozen)
`X_OBSERVABLE_KEYS = [mechanism_id, drawer_name, robot_joint_pos, tcp_pos, tcp_quat, gripper_width,
initial_mechanism_joint_pos, member]`. Anything else about the mechanism is hidden.

## 9. Theta (frozen)
Sampled dims (round-1): `grasp_offset_local_y ∈ [-0.06,0.06]` (the only bank axis), `max_pos_step ∈
[0.008,0.035]`, `pull_lead ∈ [0.03,0.14]`. In this design only `grasp_offset_local_y` varies across the bank
(fixed to the 3 bank values); `max_pos_step=0.02`, `pull_lead=0.08` are held at their exploration values.
`THETA_FIXED` (pre_grasp_clearance 0.12, approach_line_lead 0.03, reach/pull_timeout 16.0, close 1.0, settle
0.5) unchanged.
