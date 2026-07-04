# Band-edge data integrity audit v1 (read-only)

Claude B. Independent integrity gate for the 306-episode band-edge run **before** any scientific
analysis. All checks recomputed from the raw records — A's `integrity_summary.json` is NOT trusted.
Machine form: `band_edge_data_integrity_v1.json`.

## Verdict: **PASS** (scientific analysis authorized to proceed)

## Provenance (matches the authorization)
- data dir: `deployment_calibration/data/band_edge_full_306_v1/`, source commit `e447f00`, clean.
- `episodes.jsonl` sha256 = `131750e158032e53e5c8daab6e94530c70de1224f05b8fc79a7497009524c20e` ✓ (matches
  `band_edge_306_run_authorization_v1.json`).
- `config_sha256` = `2f20429b…` ✓; `science_manifest_sha256` = `e2c8216e…` ✓.

## Checks (all PASS)
| check | result |
|---|---|
| run_status COMPLETE | ✓ |
| self_check.ok | ✓ |
| 306 episodes / 306 unique ids / 306 unique (block,offset) | ✓ |
| 18 blocks, 17 offsets each | ✓ |
| offsets == frozen grid | ✓ |
| nominal_bias == 0 (all) | ✓ |
| 0 probes, 0 smoke | ✓ |
| all planned present, no unplanned | ✓ |
| config / science / episodes hashes match frozen | ✓ |
| code_commit == e447f00 (all) | ✓ |
| schema + leakage valid (all) | ✓ (B's frozen `validate_record_full`) |
| matched-block valid | ✓ |
| instrumentation complete (all 6 groups, every record) | ✓ |
| ContactSensor available (all) | ✓ |
| records/ == episodes.jsonl | ✓ |
| **identities** `actual = nominal + residual`, `eff_signed = actual + offset`, `abs_eff = |eff_signed|` | ✓ (≤1e-6/1e-9) |

## Raw summary (recomputed)
211 success / 95 failure. failure_reason: POSITION_TIMEOUT 62, HANDLE_DETACHED 33. failure_phase: APPROACH
62, PULL 33. max unintended contact force 0.0 N (all). 0 resumes recorded.

No integrity issue found → **not** `DATA_INTEGRITY_FAILURE`. The source data was read-only; nothing was
modified.
