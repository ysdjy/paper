# Band-edge runtime reliability v1

Fixes the four full-run-harness reliability blockers from Claude B's pre-run compliance audit
(`MODIFY_RUNTIME`), plus the pytest-collection minor. **No science changed, no 306 run, no confirmatory
data.** Only the run harness (`generate_calibration_bias_band_edge_v1._run_isaac`) + a reliability helper
module + tests + two reliability smokes.

## Base / branch / worktree
- Base audit commit: `630d81cde51e6ed6c2732ca91ee8da2d9f508436`
  (`origin/experiment/offline-calibration-band-edge-pre-run-audit-v1`, verdict `MODIFY_RUNTIME`)
- Branch: `experiment/runtime-calibration-band-edge-reliability-v1`
- Worktree: `/home1/banghai/Documents/IsaacLab/projects/paper_runtime_calibration_band_edge_reliability_v1`

## Files changed / added
- `band_edge_reliability_v1.py` (new) — run-status states, science-manifest SHA, atomic per-episode
  persistence, resume load/verify, dirty-source classification, hard self-check.
- `generate_calibration_bias_band_edge_v1.py` — `_run_isaac` rewritten (manifest-first, resume, fail-fast,
  self-check, dirty refusal, test hooks); new args `--stop_after_new_episodes`, `--inject_invalid_record_at`,
  `--allow_dirty`.
- `band_edge_plan_v1.py` — added a deterministic `planned_episode_id` (`b{block:02d}_{offset_id}`) per
  planned episode (id only — no science change).
- `tests/test_band_edge_plan_v1.py`, `tests/test_band_edge_instrumentation_core_v1.py` — added
  `def test_*` wrappers (pytest-discoverable).
- `tests/test_band_edge_reliability_v1.py` (new) — hard-self-check faults, dirty-source classify, atomic
  persist + resume verify, science-manifest SHA stability.

## Manifest-first & run identity (§3)
The run freezes identity before any episode: `run_id`, `config_sha256`, `science_manifest_sha256` (stable
SHA over the science content, excluding volatile git/run provenance), `source_commit`, `master_seed`,
planned `(block, offset)` keys + `planned_episode_id`s, output dir. Written to `manifest.json`,
`run_metadata.json`, `run_status.json`. Statuses: `PLANNED → IN_PROGRESS → COMPLETE | FAILED | INVALID`
(explicit transitions; partial data is never marked COMPLETE). On resume, the existing run manifest's
science/config hash must equal the frozen plan's, else `INVALID` + non-zero.

## Resume / skip-completed (§4)
- **No truncation**: the full path never `open("episodes.jsonl","w")`. Records are committed one-per-file
  under `records/<episode_id>.json` and `episodes.jsonl` is rebuilt in manifest order from committed
  records.
- **Recovery verify** (`load_and_verify_completed`): each committed record must be JSON-parseable, match
  the run's `smoke_only`, have its `planned_episode_id` + `(block,offset)` in the manifest, be unique,
  match `config_sha256` + `science_manifest_sha256`, and match the manifest block's residual / actual /
  nominal / joint-nuisance / target-jitter, and pass schema + leakage. Any inconsistency → `run_status =
  INVALID`, non-zero exit, `errors/resume_invalid.json` (no auto-overwrite, no silent repair).
- **No resample**: residual / joint nuisance / target jitter come from the frozen manifest block, never
  redrawn; completed episodes are skipped, never re-executed.

## Crash-safe persistence (§4.3)
Recommended atomic per-episode records: build → **validate** → `records/<id>.json.tmp` → `fsync` →
`os.replace` (atomic rename) → dir `fsync`. A half-written `*.tmp` is never a committed record.
`episodes.jsonl` is a derived view rebuilt from committed records.

## Fail-fast (§5)
`validate_record_full` runs **before** persistence; on failure the record is **not** written, the loop
**stops**, `run_status = FAILED`, `errors/fail_<id>.json` saves episode id + validation errors + traceback
+ completed count, and the process exits **non-zero**. Manifest/runtime mismatch (unplanned episode,
duplicate, over-count, exception) fails the same way. Test-only invalid injection is refused on the full
path.

## Dirty-source refusal (§6)
`assert_clean_source_tree_for_full_run` runs `git status --porcelain` and refuses the **full** run unless
the source tree is clean; only the run's own output dir (and `deployment_calibration/data/`) may differ,
so **resume passes when only run output changed** while any source/frozen-config/manifest change still
fails. `_classify_porcelain` is unit-tested for: clean, tracked-source-modified, frozen-config-modified,
only-output-changed. Smoke mode may run dirty (records `smoke_only=true`); the full run has no
`--allow-dirty` bypass.

## Hard self-check (§7)
After all episodes, `hard_self_check` verifies (full run): exact 306 / 18 blocks / 17 offsets each,
offsets == frozen grid, no duplicates, no probes, run-consistent smoke flag, all planned IDs present + no
unplanned, all records schema-valid + leakage-clean, matched-block valid, config/manifest hash consistent,
instrumentation required fields + `contact_sensor_available` present. Any failure → `run_status = INVALID`
+ non-zero exit; only an all-pass yields `COMPLETE` + exit 0. The self-check **controls the exit status**,
not just a report.

## Output completeness (§8)
Run dir: `manifest.json`, `run_metadata.json`, `run_status.json`, `records/`, `episodes.jsonl`,
`self_check.json`, `errors/` (on failure), `resume_log.jsonl` (resume time, source commit, manifest hash,
completed / remaining, final status). Completed valid records are never deleted.

## Test hooks (§9)
`--stop_after_new_episodes N` (test-only planned stop → `IN_PROGRESS`, exit 0) and
`--inject_invalid_record_at <planned_episode_id>` (smoke/test only) — both **forbidden on the full run**.

## pytest collection (§11)
`test_band_edge_plan_v1` and `test_band_edge_instrumentation_core_v1` now expose `def test_*`; all three
band-edge runtime test files (6 test functions) are collected. `pytest deployment_calibration/tests/`:
**99 passed** (includes B's frozen offline tests, no regression).

## Reliability smoke results (both PASS)
- **Resume smoke** (`band_edge_resume_smoke`, 2 blocks × {−0.05, 0, +0.05} = 6, smoke_only,
  `resume_smoke_result.json`): pass 1 with `--stop_after_new_episodes 3` → 3 records committed,
  `run_status=IN_PROGRESS (stopped_for_test)`; pass 2 (same run dir, resume) → **3 completed / 3 remaining**
  → skipped the 3, executed only the remaining 3 → **6 unique records, 0 duplicates**, `run_status=COMPLETE`,
  `self_check_ok=true`. `science_manifest_sha256` **identical across both passes**
  (`24fced22…`) → no resample; nuisance re-validated against the manifest on resume.
- **Forced-invalid fail-fast smoke** (`band_edge_invalid_smoke`, `--inject_invalid_record_at b01_o-0.050`,
  `invalid_record_failfast_smoke_result.json`): episode 1 valid → committed; episode 2 injected
  (`minimum_joint_limit_margin_rad = −1.0`) → **`invalid_record_written=false`**, `run_status=FAILED`
  (`failed_episode=b01_o-0.050`), **subsequent episodes not executed**, prior valid record retained,
  `errors/fail_b01_o-0.050.json` saved, process exits non-zero (generator returns 4).
- Dirty-source refusal + hard-self-check-failure are covered by pure pytest (`_classify_porcelain`,
  `hard_self_check` fault cases) — no Isaac needed.

## Authorization state
```
full_306_run_executed = false
confirmatory_generator_implemented = false
confirmatory_data_exists = false
```
No 306 run was executed and none is self-authorized. Requesting Claude B re-audit for
`AUTHORIZE_306_EXPLORATORY_RUN`.
