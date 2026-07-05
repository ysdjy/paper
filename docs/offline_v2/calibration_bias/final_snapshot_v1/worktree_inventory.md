# Worktree inventory (at Fix4 snapshot freeze)

All worktrees of `ysdjy/paper` on this machine. Each was checked with `git status --porcelain`. **No
worktree in the prereg-v4 chain (or the sampled others) has uncommitted changes** → the snapshot is not
blocked. No worktree was created or deleted by this task except the snapshot worktree.

| worktree path (basename) | branch | HEAD | clean? | prereg-v4 chain? | unmerged critical content? |
|---|---|---|---|---|---|
| paper | experiment/runtime-selection-v2 | af4a2707 | clean | no | none |
| paper_offline_calibration_prereg_v4_snapshot | **audit/…-final-snapshot-v1** | f8fedd45 | clean | THIS snapshot | none |
| paper_offline_calibration_preregistration_v4_fix4 | …-fix4 | f8fedd45 | clean | yes (source) | none |
| paper_offline_calibration_preregistration_v4_fix3 | …-fix3 | 7304530e | clean | yes | none |
| paper_offline_calibration_preregistration_v4_fix2 | …-fix2 | 48976e49 | clean | yes | none |
| paper_offline_calibration_preregistration_v4_fix1 | …-fix1 | 91a41e51 | clean | yes | none |
| paper_offline_calibration_preregistration_v4 | …-preregistration-v4 | 9a02eb5a | clean | yes | none |
| paper_offline_calibration_test_geometry_power_v1 | …-test-geometry-power-v1 | 2bf72179 | clean | yes (power cert) | none |
| paper_offline_calibration_learned_selector_power_v1 | …-learned-selector-power-v1 | 46d03afb | clean | yes | none |
| paper_offline_calibration_confirmatory_redesign_v1 | …-confirmatory-redesign-v1 | c19d434d | clean | yes | none |
| paper_offline_calibration_band_edge_analysis_v1 | …-band-edge-analysis-v1 | 93d8b168 | clean | yes | none |
| paper_offline_band_edge_authorization_v1 | …-band-edge-authorization-v1 | 0fbe68a2 | clean | related | none |
| paper_offline_band_edge_pre_run_audit_v1 | …-band-edge-pre-run-audit-v1 | 630d81cd | clean | related | none |
| paper_runtime_calibration_band_edge_reliability_v1 | …-band-edge-reliability-v1 | 04e2acc5 | clean | related (306 source) | none |
| paper_runtime_calibration_band_edge_v1 | …-band-edge-v1 | 2ac0c9f0 | clean | related | none |
| paper_offline_calibration_bias_v4 | …-offline-calibration-bias-v4 | 5f0dfdb7 | clean | no | none |
| paper_offline_calibration_bias_offline_v1 | …-offline-calibration-bias-v1 | 8b3f638e | clean | no | none |
| paper_calibration_bias_runtime_v1 | …-runtime-calibration-bias-v1 | 24919cda | clean | no | none |
| paper_integration_calibration_bias_v3 | …-integration-…-v3 | 2102563d | clean | no | none |
| paper_integration_damping_v1 | …-integration-damping-v1 | c2fd546c | clean | no | none |
| paper_offline_v2 | experiment/offline-eval-v2 | 537caf3d | clean | no | none |

## Conclusion
- The snapshot worktree (`…prereg_v4_snapshot`) is on the snapshot branch at Fix4 HEAD, clean.
- All prereg-v4-chain worktrees are clean; **no `FINAL_SNAPSHOT_BLOCKED_UNMERGED_WORKTREE_CHANGES`**.
- Sibling worktrees are prior/unrelated phases, all clean, and are **not** deleted or modified.
