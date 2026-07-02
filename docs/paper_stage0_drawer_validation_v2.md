# Stage-0 drawer validation — v2

Run: `stage0_drawer_validation_v2_20260702_210317` (contract `open_drawer_v2`, scene `paper_scene_v2`,
commit `f57d02d`, headless). 4 candidate drawers × 5 reps default params, full reset each.

Default: theta = {grasp_offset_local_y 0.0, max_pos_step 0.020, pull_lead 0.08}; g = {target 0.20, tol 0.02}.

## Result

| drawer | member | success | final positions (m) | elapsed (s) | reset stable | label ok | **STABLE** |
|---|---|---|---|---|---|---|---|
| top_drawer | cabinet | **5/5** | 0.233 / 0.232 / 0.195 / 0.234 / 0.234 | 9.3 | yes | yes | ✅ |
| middle_drawer | cabinet | **5/5** | 0.196 ×5 (σ≈1e-4) | 9.4 | yes | yes | ✅ |
| sektion_top_drawer | sektion_cabinet | **5/5** | 0.197 ×5 (σ≈1e-4) | 8.3 | yes | yes | ✅ |
| sektion_bottom_drawer | sektion_cabinet | 0/5 | 0.000 ×5 (POSITION_TIMEOUT) | 21.0 | yes | yes | ❌ excluded |

## Automated checks (all PASS)
- **test_reset_invariants_v2**: across the 5 resets of each drawer, robot-joint / TCP / drawer-joint
  max diff = **0.00** (bit-deterministic reset); init velocities = 0. Well under thresholds
  (robot 1e-3, TCP 2e-3, drawer 1e-4).
- **test_episode_labels_v2**: 20/20 — `success == (final ≥ target−tol)`; failure_reason a known token;
  `task_outcome_error == |final−target|`; overshoot ≥ 0.
- **test_scene_consistency_v2**: `validate_paper_scene_v2` PASS + **freeze cross-check** (live
  `DRAWER_TARGETS` joint/link/member == frozen `mechanism_registry.json`) PASS → no shared-config drift.
- **test_parameter_effective_v2**: 20/20 — theta echoed into the request; on success the drawer moved.

## Selected drawers for the experiment
**top_drawer, middle_drawer, sektion_top_drawer** — meet all minimum conditions (≥4/5 success,
stable reset, no manual steps, labels match physics, acceptable time ~8–9 s).

Stage-1 capability map will start with **middle_drawer + sektion_top_drawer** (tightest final-position
spread → cleanest signal; one per cabinet for later cross-mechanism generalization). top_drawer kept
as a third candidate (slightly larger overshoot variance).

## Excluded (recorded, not fixed indefinitely)
- **sektion_bottom_drawer**: 0/5, POSITION_TIMEOUT, drawer never moved (final 0.000). The handle is out
  of the arm's reliable reach from home (bottom of the floor cabinet). Reset itself is stable, so this is
  a reachability limit, not a data-pipeline problem. Excluded from round-1 per the plan; revisit only if a
  cross-mechanism experiment specifically needs it.

## Notes
- Reset is deterministic (0 diff) because the scene loads from a fixed static layout and every episode
  writes robot default joints + member drawer joints explicitly before running.
- `top_drawer` overshoots more (free drawer coasts past 0.20 to ~0.23); `middle`/`sektion_top` land at
  ~0.196–0.197 very repeatably. All are successes (≥ 0.18). The continuous final-position signal + the
  overshoot spread are exactly what the capability map needs.
- Baseline actuator damping = 3.0 on all drawers (the hidden z_secret axis, to be varied per-session in
  the pilot). Recorded per episode; verified via `read_drawer_damping_v2`.
