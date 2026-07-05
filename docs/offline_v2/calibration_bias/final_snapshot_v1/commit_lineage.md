# Commit lineage — prereg v4 evidence chain (Fix4 snapshot)

Linear history; every commit below is an **ancestor of the snapshot source** `f8fedd45…` (verified with
`git merge-base --is-ancestor`). Repository `ysdjy/paper`.

| stage | full SHA | branch | purpose | ancestor of snapshot | key files |
|---|---|---|---|---|---|
| runtime / drawer source | `742ff4db986bff52ecce140fb44f7637a2bfc5d5` | (main line) | runtime skill backend that produced the 306 run | YES | runtime skill code |
| 306 band-edge analysis | `93d8b16861861d4084ff8fa74115e10d2f0a0ffe` | experiment/offline-calibration-band-edge-analysis-v1 | read-only 306 analysis → MODIFY_RESIDUAL_OR_DESIGN (for ±0.03) | YES | band_edge_* analysis docs, `band_edge_exit_verdict_v1.json` |
| confirmatory redesign | `c19d434d05cac8dae4b4a768f3a54c572b5fe4b7` | experiment/offline-calibration-confirmatory-redesign-v1 | SELECT_CANDIDATE_BANK_REDESIGN (Design A {-0.04,0,+0.04}) | YES | confirmatory_redesign_* |
| learned-selector power | `46d03afb41f514e87e69180c7543c1b01389ce2c` | experiment/offline-calibration-learned-selector-power-v1 | REDESIGN_MODEL_OR_CLAIM (±0.03 unpowerable) | YES | learned_selector_power_* |
| **test-geometry power (POWER CERT)** | `2bf7217907f24ae54a08db71bbdcf624b110ccf0` | experiment/offline-calibration-test-geometry-power-v1 | **POWER_SUFFICIENT_FOR_PREREG_V4** (±0.035, 9/6/9) | YES | test_geometry_power_* |
| preregistration v4 | `9a02eb5ae320564bc8710380b919101d6537c02a` | experiment/offline-calibration-preregistration-v4 | PREREGISTRATION_V4_READY_FOR_INDEPENDENT_AUDIT | YES | preregistration_v4.*, confirmatory_v4_* |
| C audit #1 | `2ab39063c42f545b35beb53015c6a16402660c37` | …preregistration-v4-audit-c | MODIFY_PREREGISTRATION_V4 (4 blockers) | YES | preregistration_v4_independent_audit_c.* |
| Fix1 | `91a41e51282f716595bc3553ba500e6e2af95f40` | …preregistration-v4-fix1 | FIX1_READY (A/B/C/D) | YES | + best_single_tiebreak_invariance_v1 |
| C reaudit #2 | `21b676b749074a85ad0d9a02e47d6f712a22baa2` | …preregistration-v4-fix1-reaudit-c | MODIFY_PREREGISTRATION_V4_FIX1 (2 blockers) | YES | preregistration_v4_fix1_reaudit_c.* |
| Fix2 | `48976e4918a11406daf805f6f6451ea8272b72d5` | …preregistration-v4-fix2 | FIX2_READY (1/2) | YES | + confirmatory_v4_selection/identity, bridge invariance |
| C final reaudit #3 | `83d860693bc788c0423d9e76d1c058df32218e8b` | …preregistration-v4-fix2-final-reaudit-c | MODIFY_PREREGISTRATION_V4_FIX2 (3 blockers) | YES | preregistration_v4_fix2_final_reaudit_c.* |
| Fix3 | `7304530eb07d926b96f3e4cbd878ac50fa1d4633` | …preregistration-v4-fix3 | FIX3_READY (I/II/III) | YES | + confirmatory_v4_manifest_integrity |
| C fix3 final reaudit #4 | `06f596597731ac5fb668fc4e41563a1a85c05d74` | …preregistration-v4-fix3-final-reaudit-c | MODIFY_PREREGISTRATION_V4_FIX3 (4 blockers) | YES | preregistration_v4_fix3_final_reaudit_c.* |
| **Fix4 (snapshot source)** | `f8fedd45d2c0e2f9587ae8740c61a1b9eedb64ea` | …preregistration-v4-fix4 | **PREREGISTRATION_V4_FIX4_READY_FOR_FINAL_C_REAUDIT** | = HEAD | + test_preregistration_v4_fix4, fix4 summary |
| snapshot | (this commit; see `snapshot_manifest.json`) | audit/offline-calibration-prereg-v4-final-snapshot-v1 | freeze Fix4 for one-shot full audit | = HEAD child | docs/offline_v2/calibration_bias/final_snapshot_v1/ |

## Notes
- Power certification (`2bf7217`) carried through unchanged across Fix1–Fix4 (each fix commit records
  `power_recertification_required = false`).
- C's four audit commits are on separate `…-reaudit-c` / `…-audit-c` branches but are linear ancestors of
  each subsequent Fix (each Fix branched from the corresponding C audit commit).
- The C audit reports (`preregistration_v4_*_reaudit_c.{md,json}`) and their test files are **audit-only**
  historical evidence and are present unmodified in the snapshot tree.
