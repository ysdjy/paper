# Band-edge continuous outcomes v1 (read-only, exploratory secondary)

Claude B. Pre-registered exploratory secondary description of the continuous outcomes vs success / failure
(and, by extension, |eff|). No large post-hoc model search — only the block-aware summaries the protocol
allows. Machine form: `_band_edge_analysis_bundle_v1.json`.

## Success vs failure means (306 episodes)
| field | success mean | failure mean |
|---|---|---|
| task_outcome_error (m) | 0.011 | 0.174 |
| skill_elapsed_time (s) | 9.42 | 23.75 |
| true_handle_error_at_close_3d (m) | 0.0072 | 0.0230 |
| true_handle_error_at_close_local_y (m) | 0.00022 | 0.00191 |
| gripper_width_at_close (m) | 0.0111 | 0.0088 |

## Reading (consistent with a grasp-offset compensation edge)
- **task_outcome_error** and **skill_elapsed_time** separate cleanly: successes open the drawer to target
  (~0.011 m error, ~9.4 s); failures do not (~0.17 m error, ~23.8 s — many APPROACH timeouts run long).
- **true_handle_error_at_close_3d** (the TRUE, unbiased handle-vs-TCP error at CLOSE→PULL) is ~3× larger in
  failures (0.023 vs 0.0072 m) — the direct signature of an off-centre grasp near/over the edge, and the
  reason the handle detaches during PULL.
- **gripper_width_at_close** is similar in both (~0.009–0.011 m) — failures are **not** empty grasps (the
  gripper did close on something); combined with the larger handle error this confirms an **off-centre
  grasp**, not a missed/empty grasp.

These continuous outcomes are reported as exploratory context; the primary boundary result and the exit
verdict rest on the success-label analysis (`band_edge_characterization_analysis_v1.md`), not on these.
