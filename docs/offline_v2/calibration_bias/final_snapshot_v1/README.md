# Preregistration v4 — Fix4 final snapshot (final_snapshot_v1)

This branch `audit/offline-calibration-prereg-v4-final-snapshot-v1` **freezes the Fix4 repository state** of
the offline calibration-bias preregistration v4 work for a **single one-shot full audit**.

- Repository: `ysdjy/paper`
- Source Fix4 commit: `f8fedd45d2c0e2f9587ae8740c61a1b9eedb64ea`
- Active protocol status: `PREREGISTRATION_V4_FIX4_READY_FOR_FINAL_C_REAUDIT`
- Power certification: `2bf7217907f24ae54a08db71bbdcf624b110ccf0` (POWER_SUFFICIENT_FOR_PREREG_V4, ±0.035, 9/6/9)

## What this is / is NOT
- This directory holds **snapshot metadata only**. The active code, tests, and protocol remain at their
  original paths (`deployment_calibration/offline_v2/calibration_bias/`,
  `deployment_calibration/tests/offline_v2/calibration_bias/`, `docs/offline_v2/calibration_bias/`). The
  repository is **not** copied into this directory, and no tar/zip is committed.
- This is **NOT** a scientific GO. It does **not** authorize the generator, formal manifest generation,
  confirmatory data, or a confirmatory run. Claude A remains paused.

## Audit scope & rule
- The next step is exactly **one** full-repository audit that freezes a **complete** issue list in one pass.
- **No** more "find one → fix one → re-audit" loops. Observations found while building this snapshot are
  recorded in `known_issues_at_freeze.md` and were **not** fixed here.

## Evidence chain
Full lineage in `commit_lineage.md`: 306 source → band-edge analysis → redesign → learned-selector power →
**test-geometry power (cert)** → preregistration v4 → C audit → Fix1 → C → Fix2 → C → Fix3 → C → **Fix4**.
Every commit is a linear ancestor of the snapshot.

## Metadata files
| file | contents |
|---|---|
| `snapshot_manifest.json` | machine summary + final snapshot commit SHA |
| `active_file_map.json` | the 13 active files + evidence/historical, with sha256 + git blob sha |
| `repository_file_inventory.txt` | `git ls-files` (1154 tracked files) |
| `repository_tree.txt` | `find` tree (excludes `./.git`) |
| `tracked_file_hashes.sha256` | sha256 of every tracked file, sorted by path |
| `git_snapshot_state.md` / `git_log_graph_400.txt` | git status/branch/HEAD/remotes/log |
| `worktree_inventory.md` | all worktrees + clean/dirty + unmerged check |
| `commit_lineage.md` | the evidence-chain commit map |
| `external_artifact_manifest.json` | 306 data / capability maps / env (what is / isn't in git) |
| `environment_snapshot.md` | python/torch/numpy/OS/GPU text captures |
| `test_report.md` | active B-owned = 244 passed; C audit files classified separately |
| `global_reference_scan.md` | keyword scan, classified active-valid / potential-conflict / historical |
| `known_issues_at_freeze.md` | observations for the one-shot audit (A–D) |

## How to verify hashes
- 306 raw data: `sha256sum deployment_calibration/data/band_edge_full_306_v1/episodes.jsonl` →
  `131750e1…9524c20e`; dir hash `718e0a70…c68844f` (algorithm in `external_artifact_manifest.json`).
- Tracked files: compare against `tracked_file_hashes.sha256`
  (`sha256 = int(sha256(path||NUL||...))` per line: `<sha256>␠␠<relpath>`).
- Active files: `active_file_map.json` gives each file's sha256 and `git hash-object` blob sha.

## Constraints honored
No generator; no formal manifest instance; no checkpoint; no runtime/confirmatory data; no run
authorization. 306 data, runtime, `models_v2`, capability map, power results, C historical audits, and the
original band-edge verdict are untouched.
