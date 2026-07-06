# Scientific pilot — 306 data inventory (EXPLORATORY / PILOT — NOT CONFIRMATORY)

Dataset: `band_edge_full_306_v1/episodes.jsonl` — sha256 `131750e1…09524c20e` (306 episodes).
Machine-readable copy: `scientific_pilot_data_inventory.json`.

## Structure
- **306 episodes = 18 blocks × 17 offsets**, exactly one trial per (block, offset). The grid is complete (18×17).
- **Every episode is `episode_role="candidate"` — there is NO native probe episode and NO native
  "1 probe + 3 candidate" session design.** The intended confirmatory session structure is absent here; this run
  is a fine-grained **capability-map / band-edge sweep**.
- Offsets: `-0.05, -0.04, -0.035, -0.03, -0.025, -0.02, -0.015, -0.01, 0.0, 0.01, 0.015, 0.02, 0.025, 0.03,
  0.035, 0.04, 0.05` (grasp_offset_local_y).

## Hidden state (labels only — never model inputs)
- **All 18 blocks share `nominal_bias_y = 0.0`.** The only hidden-state variation is the per-block
  `residual_bias_y ∈ [−0.0067, +0.0092]` (≈ ±9 mm), drawn per block.
- `secret_deployment_state` = {nominal_bias_y, residual_bias_y, actual_bias_y}; also hidden:
  `eff_signed`, `abs_eff`, `true_handle_error_at_close_local_y/3d`, `true_handle_pose_at_close`. All used only as
  analysis labels; the leakage guard (`scientific_pilot_leakage_check.json`) blocks them from features.

## Fields present (per episode)
`block_id`, `block_seed`, `planned_episode_id`, `theta.grasp_offset_local_y` (the action), `g.target_open_position`,
`g.target_tolerance`, `x` (robot_joint_pos, tcp_pos/quat, gripper_width, initial_mechanism_joint_pos),
`y` (success, task_outcome_error, phase_goal_error, handle_relative_error, overshoot, skill_elapsed_time,
phase_durations, handle_detached, command_tracking_error), block-level `nuisance_*`, and the hidden
`secret_deployment_state`.

## Sufficiency verdict
**Sufficient** for (a) block-level **oracle-headroom** analysis and (b) **probe-EMULATED** identifiability /
selection — where the offset=0.0 episode of each block is designated as the "probe" and its execution outcome
`y` is the probe history, with the probe offset excluded from the candidate-selection set.

**Two structural limitations** (not missing fields, but design mismatch vs. the confirmatory hypothesis):
1. No native probe+3-candidate session design (probe must be emulated).
2. All blocks are `nominal_bias_y = 0`, so hidden-state variation is limited to small residual draws
   (±9 mm) — well inside the grasp success tolerance band. This is the root cause of zero oracle headroom
   (see the report). No fields are missing for the pilot analyses; the limitation is the **bias range**, not
   the schema.

No new simulation was run; no confirmatory seeds were touched.
