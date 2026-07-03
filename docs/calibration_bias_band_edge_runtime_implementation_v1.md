# Band-edge runtime implementation v1

Runtime (Claude A) implementation of Claude B's FROZEN band-edge characterization protocol
(`docs/offline_v2/calibration_bias/band_edge_characterization_protocol_v1.md` +
`offline_v2/calibration_bias/band_edge_characterization_config_v1.json` + schema + learned-selector power
plan). **This phase implements the generator + six instrumentation groups + collision sensor + validators
+ smokes only.** No full 306 run, no confirmatory generator, no confirmatory data. Isaac was launched only
for the sensor/instrumentation/clear-zone smokes.

## Base / branch / worktree
- Base commit (frozen): `5f0dfdb74cd919ce69701e05cbf451fca4170365` (`origin/experiment/offline-calibration-bias-v4`)
- Branch: `experiment/runtime-calibration-band-edge-v1`
- Worktree: `/home1/banghai/Documents/IsaacLab/projects/paper_runtime_calibration_band_edge_v1`
- Frozen config SHA256: `2f20429b5273cb7d…` (`band_edge_characterization_config_v1.json`)

## Implemented files
- `deployment_calibration/data_generation/band_edge_plan_v1.py` — 306 plan/manifest (18×17, no probes),
  residual via B's `draw_block_residuals` (no second algorithm), per-block residual+nuisance shared across
  17 offsets, randomized block + offset order, resume-safe manifest with all provenance.
- `deployment_calibration/data_generation/band_edge_validators_v1.py` — plan/count, matched-block, leakage
  (`assert_residual_not_in_x` + `FORBIDDEN_X_SUBSTRINGS` + `OBSERVABLE_NUISANCE_KEYS` gate DISABLED), smoke
  exclusion, duplicate rejection, no-probes; schema via B's `validate_record`.
- `band_edge_instrumentation_core_v1.py` — PURE cores (joint margin, IK-failure counter, two clamp
  counters, failure phase, CLOSE→PULL snapshot). Unit-tested without Isaac.
- `band_edge_instrumentation_runtime_v1.py` — `InstrumentedIKAdapter` (IK-failure + clamp counting around
  `solve()`, no behaviour change) + `CollisionMonitor` (ContactSensor net force, non-finger links).
- `band_edge_runtime_v1.py` — instrumented episode runner (mirrors verified `run_drawer_episode_v2`);
  injection via `handle_calibration_bias_v1.biased_spec`; TRUE (unbiased) handle for the close snapshot.
- `generate_calibration_bias_band_edge_v1.py` — modes `manifest_only` (no Isaac), `smoke`, `full` (GATED
  — refuses without explicit approval; never authorized here).
- `band_edge_contact_smoke_v1.py` — contact-sensor validation smoke (clean baseline + positive control).
- Configs: `calibration_bias_band_edge_clearzone_smoke_v1.yaml` (2 blocks × {−0.05, 0, +0.05}).
- Tests: `tests/test_band_edge_plan_v1.py`, `tests/test_band_edge_instrumentation_core_v1.py`.

## Residual implementation source
Reuses `offline_v2/calibration_bias/residual_nuisance.draw_block_residuals` /
`ResidualNuisanceConfig(sigma=0.005, lo=−0.01, hi=+0.01)` verbatim — one residual per block, shared by all
17 offsets; NO second residual algorithm. `actual_bias_y = nominal(0) + residual_bias_y`. Test asserts the
runtime residuals equal `draw_block_residuals(...)`.

## Instrumentation mapping (schema field → source)
| group | fields | source |
|---|---|---|
| 5.1 joint margin | rad / normalized / joint_index / phase (+ negative flag) | per-step `robot.data.joint_pos[0:7]` vs `adapter._joint_lower/_upper` (soft limits) |
| 5.2 IK failure | solve / failure / max-consecutive / phase counts | `InstrumentedIKAdapter.solve().success` |
| 5.3 clamps | step-clamp vs limit-clamp (never merged) | `detect_clamps(q_curr, q_des, limits, max_joint_step)` on each `solve()` |
| 5.4 failure phase | failure_phase / failure_transition | last active state before FAILED from `skill.runtime.history` |
| 5.5 close snapshot | true 3D / local-Y / tcp_pose / true_handle_pose | at the CLOSE_GRIPPER→PULL step; TRUE (unbiased) handle from link pose + `spec.handle_local_pos` |
| 5.6 gripper at close | width / command | same control step |
| 6 collision | max force / frames / first phase / by phase / available | `CollisionMonitor` |

Absent snapshot → all 5.5/5.6 fields `null`, `close_snapshot_available=false`, `close_snapshot_reason` set.

## Contact backend
`isaaclab.sensors.ContactSensor` installed by `enable_collision_monitor=True` (`scene['robot_contact']`,
`data.net_forces_w` per robot link). Unintended contact = net force on **non-finger** links (arm + hand);
the two finger links are excluded (their contact is the intended finger↔handle grasp). Equivalence: in this
scene the only bodies the arm/hand can touch are the cabinet/drawer, so non-finger link force ==
arm/hand-vs-cabinet unintended contact. `contact_sensor_backend` records this string per episode.

## Joint-margin range note (transparent, NOT a silent substitution)
The frozen `minimum_joint_limit_margin_rad` field is declared range `[0, joint_range]` and B's
`validate_record` enforces `>= 0`. A joint can momentarily sit a hair past a **soft** limit (~1e-5 rad),
giving a tiny negative raw margin. We therefore report the frozen field **clamped to its declared
`[0, joint_range]` range**, and preserve the true excursion via two extra audit fields:
`minimum_joint_limit_margin_rad_signed` (true signed value) and `joint_limit_margin_negative_flag` (the
task's required negative flag). No information is lost; the frozen schema validator passes.

## Smoke results (all PASS)
- **Contact-sensor smoke** (`band_edge_contact_smoke_v1_20260704_000958`): `sensor_available=true`;
  clean baseline **0.000 N** vs positive control (hand driven into cabinet body) **240.08 N**; gap 240 N;
  `discriminates=true`. Backend: `ContactSensor(robot_contact.net_forces_w, non-finger links)`.
- **Clear-zone instrumentation smoke** (`band_edge_smoke_v1_20260704_001540`, 2 blocks × {−0.05, 0, +0.05}
  = 6 eps, `smoke_only=true`, 3.25 min): `all_records_valid=true`, `matched_block_ok=true` (single
  `actual_bias` per block, initial-x spread 0.00 — only offset varies), `duplicate_ok=true`,
  `no_probes_ok=true`, `contact_sensor_available=true`. All six instrumentation groups present on every
  record. Physical sanity: offset 0 → success (close snapshot captured, TRUE-handle error ~0.005 m);
  offset ±0.05 → fail (off-centre −0.05 reached CLOSE→PULL with TRUE-handle error 0.056 m — the off-centre
  diagnostic; +0.05 failed before CLOSE→PULL → snapshot null + reason). Core edge points
  ±0.025/0.030/0.035 deliberately NOT used; no edge was fit; the science protocol is untouched.
- Smoke data carry `smoke_only=true` (`dirty_worktree=true` is fine — these are dev-time diagnostics,
  excluded from any formal analysis by `exclude_smoke`). The full 306 run would be generated from a clean
  commit.

## Test results
- `test_band_edge_plan_v1` PASS; `test_band_edge_instrumentation_core_v1` PASS.
- B's frozen `tests/offline_v2/calibration_bias/` regression: 59 passed.
- `manifest_only` → 306 planned, no probes, `validate_plan` OK; `full` mode refuses without approval.

## Authorization flags
- `full_band_edge_run_authorized = false`
- `confirmatory_generator_implemented = false`
- `confirmatory_data_exists = false`
